from itertools import permutations

class Game:
    
    def __init__(self, utility):
        self.utility = dict(utility)
        self.strategies = sorted({ strategy
            for outcome in utility.keys()
            for strategy in outcome
        })

    def u(self, player, p_strat, opp_strat):
        if player == 0:
            return self.utility[p_strat, opp_strat][player]
        else:
            return self.utility[opp_strat, p_strat][player]

    def is_strongly_dominated(self, player, s_i, s_mark):
        return all(self.u(player,s_mark, s_opp) > self.u(player, s_i,s_opp)  for s_opp in self.strategies)

    def is_strongly_dominant_strategy(self,player, s_star):
        return all(self.is_strongly_dominated(player, s_i, s_star) for s_i in self.strategies if s_i != s_star)

    def has_strongly_dominant_strategy(self, player):
        return any(self.is_strongly_dominant_strategy(player, s) for s in self.strategies)

    def is_weakly_dominated(self, player, s_i, s_mark):
        """ s_i is weakly dominated by s_mark"""
        return all(self.u(player,s_mark, s_opp) >= self.u(player, s_i,s_opp)  for s_opp in self.strategies) and any(self.u(player,s_mark, s_opp) > self.u(player, s_i,s_opp)  for s_opp in self.strategies)

    def is_weakly_dominant_strategy(self, player, s_star):
        return all(self.is_weakly_dominated(player, s_i, s_star) for s_i in self.strategies if s_i != s_star)

    def has_weakly_dominant_strategy(self, player):
        return any(self.is_weakly_dominant_strategy(player, s) for s in self.strategies)

    def is_very_weakly_dominated(self, player, s_i, s_mark):
        """ s_i is weakly dominated by s_mark"""
        return all(self.u(player,s_mark, s_opp) >= self.u(player, s_i,s_opp)  for s_opp in self.strategies)

    def is_very_weakly_dominant_strategy(self, player, s_star):
        return all(self.is_very_weakly_dominated(player, s_i, s_star) for s_i in self.strategies if s_i != s_star)

    def has_very_weakly_dominant_strategy(self, player):
        return any(self.is_very_weakly_dominant_strategy(player, s) for s in self.strategies)

    def pure_nash(self):
        nash_eq = []
        for a,b in self.utility.keys():
            #test for player 0
            p0 = all(self.u(0,a, b) >= self.u(0,s_i, b) for s_i in self.strategies)
            p1 = all(self.u(1,b, a) >= self.u(1,s_i, a) for s_i in self.strategies)
            if p0 and p1:
                nash_eq.append((a,b))
        return nash_eq
    
    def strongly_dominant_equilibrium(self):
        strong_eq = []
        for a,b in self.utility.keys():
            #test for player 0
            p0 = self.is_strongly_dominant_strategy(0,a)
            p1 = self.is_strongly_dominant_strategy(1,b)

            if p0 and p1:
                strong_eq.append((a,b))
        return strong_eq

    def weakly_dominant_equilibrium(self):
        weakly_eq = []
        for a,b in self.utility.keys():
            #test for player 0
            p0 = self.is_weakly_dominant_strategy(0,a)
            p1 = self.is_weakly_dominant_strategy(1,b)

            if p0 and p1:
                weakly_eq.append((a,b))
        return weakly_eq

    def very_weakly_dominant_equilibrium(self):
        very_weakly_eq = []
        for a,b in self.utility.keys():
            #test for player 0
            p0 = self.is_very_weakly_dominant_strategy(0,a)
            p1 = self.is_very_weakly_dominant_strategy(1,b)

            if p0 and p1:
                very_weakly_eq.append((a,b))
        return very_weakly_eq

    def print_equilibrium(self):
        print(f"Strongly dominant equilibrium: {self.strongly_dominant_equilibrium()}")
        print(f"Weakly dominant equilibrium: {self.weakly_dominant_equilibrium()}")
        print(f"Very weakly dominant equilibrium: {self.very_weakly_dominant_equilibrium()}")
        print(f"Nash equilibrium: {self.pure_nash()}")

    def maxmin_value(self,player):
        return max( min(  self.u(player, p_strat, opp_strat) for opp_strat in self.strategies )  for p_strat in self.strategies )

    def maxmin_strategies(self, player):
        v = self.maxmin_value(player)
        return [p_strat for p_strat in self.strategies if min(  self.u(player, p_strat, opp_strat) for opp_strat in self.strategies ) == v]

    def minmax_value(self,player):
        return min( max( self.u(player,p_strat,opp_strat) for p_strat in self.strategies) for opp_strat in self.strategies  )

    def minmax_strategies(self, player):
        """minmax strategies against player 
        """
        v = self.minmax_value(player)
        return [opp_strat for opp_strat in self.strategies if max(self.u(player, p_strat, opp_strat) for p_strat in self.strategies) == v ]

def tests():
    g = Game({
        ('A','A'): [-2,-2], ('A','B'): [-10,-1],
        ('B','A'): [-1,-10], ('B','B'): [-5,-5],
    })
   
    assert g.is_strongly_dominated(0,'A','B')
    assert g.is_strongly_dominated(1,'A','B')
    assert not g.is_strongly_dominated(1,'B','A')

    assert g.is_strongly_dominant_strategy(0,'B')
    assert not g.is_strongly_dominant_strategy(0,'A')

    assert g.has_strongly_dominant_strategy(0)
    assert g.has_strongly_dominant_strategy(1)

    g = Game({
        ('A','A'): [-2,-2], ('A','B'): [-10,-2],
        ('B','A'): [-2,-10], ('B','B'): [-5,-5],
    })

    assert not g.has_strongly_dominant_strategy(0)
    assert not g.has_strongly_dominant_strategy(1)

    assert g.is_weakly_dominated(0,'A','B')
    assert g.is_weakly_dominant_strategy(0,'B')
    assert not g.is_weakly_dominant_strategy(0,'A')

    assert g.weakly_dominant_equilibrium() == [('B','B')]

    #example 6.1
    g = Game({
        ('A','A'): [2,1], ('A','B'): [0,0],
        ('B','A'): [0,0], ('B','B'): [1,2],
    })
    assert g.pure_nash() == [('A','A'),('B','B')]

    assert not g.strongly_dominant_equilibrium()

    #example 6.2
    g = Game({
        ('A','A'): [-2,-2], ('A','B'): [-10,-1],
        ('B','A'): [-1,-10], ('B','B'): [-5,-5],
    })
    assert g.pure_nash() == [('B','B')]

    g = Game({
        ('A','A'): [6,6], ('A','B'): [0,8],
        ('B','A'): [8,0], ('B','B'): [3,3],
    })
    assert g.pure_nash() == [('B','B')]

    #maxmin
    g = Game({
        ('A','A'): [4,1], ('A','B'): [0,4],
        ('B','A'): [1,5], ('B','B'): [1,2],
    })
    assert g.maxmin_value(0) == 1
    assert g.maxmin_value(1) == 2

    g = Game({
        ('A','A'): [4,1], ('A','B'): [0,4],
        ('B','A'): [1,5], ('B','B'): [1,1],
    })
    assert g.maxmin_strategies(0) == ['B']
    assert g.maxmin_strategies(1) == ['A','B']


def exercise_6_9():
    #exercise 6.9.a)
    outcomes = [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
    games = lambda: (
        Game({ k:v for k,v in zip(outcomes, zip(comb[::2], comb[1::2]))})
        for comb in permutations([0,1]*8,8)
    )

    for g in games():
        if len(g.pure_nash()) == 1:
            if g.pure_nash()[0] not in g.weakly_dominant_equilibrium():
                print(g.utility, " has unique nash that is not weakly dominant.")
                break
    else:
        print("could not find solution.")
        

    #exercise 6.9.b)
    for g in games():
        if len(g.pure_nash()) == 1:
            if g.pure_nash()[0] in g.weakly_dominant_equilibrium() and g.pure_nash()[0] not in g.strongly_dominant_equilibrium():
                print(g.utility, " has unique nash that is weakly dominant, but not strongly dominant")
                break
    else:
        print("could not find solution.")

    #exercise 6.9.c)
    for g in games():
        if g.weakly_dominant_equilibrium() and g.pure_nash():
            if any( i not in  g.weakly_dominant_equilibrium() and i not in g.weakly_dominant_equilibrium() for i in g.pure_nash()):
                print(g.utility, " has nash and weak ")
                break
    else:
        print("could not find solution.")

def exercise_6_3():
    g = Game({
        ('A','A'): [0,1], ('A','B'): [1,1],
        ('B','A'): [1,1], ('B','B'): [1,0],
    })

    print(f"pure nash in : {g.pure_nash()}")
    for p in [0,1]:
        print(f'minmax_value for player {p} is {g.maxmin_value(p)}')

    for p in [0,1]:
        print(f'minmax_strategy for player {p} is {g.maxmin_strategies(p)}')

    for p in [0,1]:
        print(f'maxmin_value for player {p} is {g.minmax_value(p)}')

    for p in [0,1]:
        print(f'maxmin_strategy for player {p} is {g.minmax_strategies(p)}')


if __name__ == '__main__':        
    
    tests()    

    exercise_6_3()
    #exercise_6_9()

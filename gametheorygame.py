from itertools import combinations_with_replacement, product


class Game:
    
    def __init__(self):
        self.strategies = ['A','B']
        self.utility = {('A','A') : [-2,-2], ('A','B'): [-10,-1], ('B','A') : [-1,-10], ('B','B'): [-5,-5]}

    def u(self, player, p_strat, opp_strat):
        if player == 0:
            return self.utility[p_strat, opp_strat][player]
        else:
            return self.utility[opp_strat, p_strat][player]

    def set_utility(self, utilities):
        """
            list of 4 list of 2 representing (A,A), (A,B), (B,A), (B,B)
        """
        
        self.utility[('A','A')] = utilities[0]
        self.utility[('A','B')] = utilities[1]
        self.utility[('B','A')] = utilities[2]
        self.utility[('B','B')] = utilities[3]

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
        for a,b in product(self.strategies,repeat=2):
            #test for player 0
            p0 = all(self.u(0,a, b) >= self.u(0,s_i, b) for s_i in self.strategies)
            p1 = all(self.u(1,b, a) >= self.u(1,s_i, a) for s_i in self.strategies)
            if p0 and p1:
                nash_eq.append((a,b))
        return nash_eq
    
    def strongly_dominant_equilibrium(self):
        strong_eq = []
        for a,b in product(self.strategies,repeat=2):
            #test for player 0
            p0 = self.is_strongly_dominant_strategy(0,a)
            p1 = self.is_strongly_dominant_strategy(1,b)

            if p0 and p1:
                strong_eq.append((a,b))
        return strong_eq

    def weakly_dominant_equilibrium(self):
        weakly_eq = []
        for a,b in product(self.strategies,repeat=2):
            #test for player 0
            p0 = self.is_weakly_dominant_strategy(0,a)
            p1 = self.is_weakly_dominant_strategy(1,b)

            if p0 and p1:
                weakly_eq.append((a,b))
        return weakly_eq

    def very_weakly_dominant_equilibrium(self):
        very_weakly_eq = []
        for a,b in product(self.strategies,repeat=2):
            #test for player 0
            p0 = self.is_very_weakly_dominant_strategy(0,a)
            p1 = self.is_very_weakly_dominant_strategy(1,b)

            if p0 and p1:
                very_weakly_eq.append((a,b))
        return very_weakly_eq

    def maxmin_value(self,player):
        return max( min(  self.u(player, p_strat, opp_strat) for opp_strat in self.strategies )  for p_strat in self.strategies )

    def maxmin_strategies(self, player):
        v = self.maxmin_value(player)
        return [p_strat for p_strat in self.strategies if min(  self.u(player, p_strat, opp_strat) for opp_strat in self.strategies ) == v]

def tests():
    g = Game()
   
    assert g.is_strongly_dominated(0,'A','B')
    assert g.is_strongly_dominated(1,'A','B')
    assert not g.is_strongly_dominated(1,'B','A')

    assert g.is_strongly_dominant_strategy(0,'B')
    assert not g.is_strongly_dominant_strategy(0,'A')

    assert g.has_strongly_dominant_strategy(0)
    assert g.has_strongly_dominant_strategy(1)

    g.set_utility([[-2,-2],[-10,-2],[-2,-10],[-5,-5]])

    assert not g.has_strongly_dominant_strategy(0)
    assert not g.has_strongly_dominant_strategy(1)

    assert g.is_weakly_dominated(0,'A','B')
    assert g.is_weakly_dominant_strategy(0,'B')
    assert not g.is_weakly_dominant_strategy(0,'A')

    assert g.weakly_dominant_equilibrium() == [('B','B')]

    #example 6.1
    g.set_utility([[2,1],[0,0],[0,0],[1,2]])
    assert g.pure_nash() == [('A','A'),('B','B')]

    assert not g.strongly_dominant_equilibrium()

    #example 6.2
    g.set_utility([[-2,-2],[-10,-1],[-1,-10],[-5,-5]])
    assert g.pure_nash() == [('B','B')]

    g.set_utility([[6,6],[0,8],[8,0],[3,3]])
    assert g.pure_nash() == [('B','B')]

    #maxmin
    g.set_utility([[4,1],[0,4],[1,5],[1,2]])
    assert g.maxmin_value(0) == 1
    assert g.maxmin_value(1) == 2

    g.set_utility([[4,1],[0,4],[1,5],[1,1]])
    assert g.maxmin_strategies(0) == ['B']
    assert g.maxmin_value(1) == ['A']
    

def exercise_6_9():
    #exercise 6.9.a)
    g = Game()
    for comb in combinations_with_replacement(range(1,8),8):
        g.set_utility([comb[i: i+2] for i in range(0, len(comb), 2)])

        if len(g.pure_nash()) == 1:
            if g.pure_nash()[0] not in g.weakly_dominant_equilibrium():
                print(g.utility, " has unique nash that is not weakly dominant.")
                break
    else:
        print("could not find solution.")
        

    #exercise 6.9.b)
    for comb in combinations_with_replacement(range(1,9),8):
        g.set_utility([comb[i: i+2] for i in range(0, len(comb), 2)])

        if len(g.pure_nash()) == 1:
            if g.pure_nash()[0] in g.weakly_dominant_equilibrium() and g.pure_nash()[0] not in g.strongly_dominant_equilibrium():
                print(g.utility, " has unique nash that is weakly dominant, but not strongly dominant")
                break
    else:
        print("could not find solution.")

    #exercise 6.9.c)
    for comb in combinations_with_replacement(range(1,9),8):
        g.set_utility([comb[i: i+2] for i in range(0, len(comb), 2)])

        if g.weakly_dominant_equilibrium() and g.pure_nash():
            if any( i not in  g.weakly_dominant_equilibrium() and i not in g.weakly_dominant_equilibrium() for i in g.pure_nash()):
                print(g.utility, " has nash and weak ")
                break
    else:
        print("could not find solution.")

def exercise_6_3():
    g = Game()

    g.set_utility([[0,1],[1,1],[1,1],[1,0]])

    print(g.pure_nash())



if __name__ == '__main__':        
    
    tests()    

    exercise_6_3()
    #exercise_6_9()
